import React from "react"
import PropTypes from "prop-types"
import Recipe from "./recipe"
import { Container, Grid } from '@material-ui/core'

const RecipeBoard = ({recipes}) => {
  return (
    <Container maxWidth="md">
      <Grid container spacing={4}>
      {recipes.map(({ node }, index) => (
         <Grid item key={index} xs={12} sm={6} md={4}>
          <Recipe title={node.recipe.name} url={node.recipe.url} category={node.recipe.category} desc={node.recipe.desc}/>
         </Grid>
      ))}
      </Grid>
    </Container>
  ) 
}

RecipeBoard.propTypes = {
  recipes: PropTypes.arrayOf(PropTypes.object).isRequired,
}
export default RecipeBoard
